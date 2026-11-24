import asyncio

import discord

from nezubot_discrod.messages import split_message
from nezubot_discrod.policy import PolicyViolation
from nezubot_discrod.sandbox import accepting_work, truncate_output


class ApprovalView(discord.ui.View):
    def __init__(self, service, proposal):
        super().__init__(timeout=None)
        self.service = service
        self.proposal = proposal

    async def interaction_check(self, interaction):
        if (
            interaction.user.id != self.service.settings.owner_id
            or interaction.guild_id != self.service.settings.guild_id
        ):
            await interaction.response.send_message("オーナーのみ操作できます。", ephemeral=True)
            return False
        return True

    async def decide(self, interaction, approve):
        if not await self.interaction_check(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        try:
            if approve:
                await interaction.followup.send(
                    "承認を受け付けました。マージ・更新後に再起動します。", ephemeral=True
                )
                await self.service.approve(interaction.user.id, self.proposal.number)
            else:
                await self.service.reject(interaction.user.id, self.proposal.number)
                await interaction.followup.send(
                    "PR を閉じ、ブランチを削除しました。", ephemeral=True
                )
        except Exception:
            await interaction.followup.send(
                "操作に失敗しました。PR・checkout の状態を確認して再試行してください。",
                ephemeral=True,
            )
            return
        for item in self.children:
            item.disabled = True
        await interaction.message.edit(view=self)
        self.stop()

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.success)
    async def approve_button(self, interaction, button):
        await self.decide(interaction, True)

    @discord.ui.button(label="Reject", style=discord.ButtonStyle.danger)
    async def reject_button(self, interaction, button):
        await self.decide(interaction, False)


async def handle_improve(interaction, request, service):
    if not accepting_work():
        await interaction.response.send_message("再起動準備中です。", ephemeral=True)
        return
    if (
        interaction.user.id != service.settings.owner_id
        or interaction.guild_id != service.settings.guild_id
    ):
        await interaction.response.send_message("オーナーのみ利用できます。", ephemeral=True)
        return
    await interaction.response.send_message(
        "自己改善の依頼を受け付けました。結果をこのチャンネルに投稿します。", ephemeral=True
    )
    await run_improvement(service, request, interaction.channel, interaction.user.id)


def improvement_callback(service, message, tasks, bot_user_id):
    async def start(_request):
        if message.author.id != service.settings.owner_id:
            return "Only the owner can request self-improvement."
        if message.guild is None or message.guild.id != service.settings.guild_id:
            return "自己改善は指定サーバーで依頼してください。DM からは開始できません。"
        if not accepting_work():
            return "再起動準備中のため自己改善を開始できません。"
        if tasks or service.lock.locked() or service.pending is not None:
            return "自己改善の処理または承認待ちの PR があります。"
        request = (
            message.content.replace(f"<@{bot_user_id}>", "")
            .replace(f"<@!{bot_user_id}>", "")
            .strip()
        )
        if not request or len(request) > 8000:
            return "依頼は元のメッセージに 1〜8000 文字で入力してください。"
        task = asyncio.create_task(run_improvement(
            service, request, message.channel, message.author.id
        ))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return "Started; the PR link and Approve/Reject buttons will be posted in this channel."

    return start


async def run_improvement(service, request, channel, user_id):
    try:
        proposal = await service.propose(user_id, request)
    except PolicyViolation as error:
        for part in split_message(str(error)):
            await channel.send(part, allowed_mentions=discord.AllowedMentions.none())
        return
    except Exception as error:
        await channel.send(
            "自己改善を停止しました。\n" + truncate_output(str(error), 800),
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return
    await channel.send(
        f"レビュー待ち: {proposal.url}\n```\n{truncate_output(proposal.stat, 600)}\n```",
        view=ApprovalView(service, proposal),
        allowed_mentions=discord.AllowedMentions.none(),
    )
